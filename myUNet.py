import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups=8, epsilon=1e-5):
        super(GroupNormalization, self).__init__()
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=[input_shape[-1]], initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta', shape=[input_shape[-1]], initializer=tf.zeros_initializer(),
                                    trainable=True)
        self.batch_shape = [1, 1, 1, 1]
        self.batch_shape[0] = -1
        self.batch_shape[3] = input_shape[-1] // self.groups
        self.group_shape = [-1, input_shape[1], input_shape[2], self.groups, self.batch_shape[3]]

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        group_shape = tf.concat([self.group_shape[0:3], [self.groups, inputs_shape[3] // self.groups]], axis=0)
        groups = tf.reshape(inputs, group_shape)
        mean, var = tf.nn.moments(groups, [1, 2, 4], keepdims=True)
        groups = (groups - mean) / tf.sqrt(var + self.epsilon)
        groups = tf.reshape(groups, tf.shape(inputs))
        outputs = self.gamma * groups + self.beta
        return outputs


class AttentionBlock(layers.Layer):
    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)
        self.norm = GroupNormalization()
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)
        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])
        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])
        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="uniform"
    )


def residual_block(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t, p = inputs
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)
        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[
            :, None, None, :
        ]
        pemb = activation_fn(p)
        pemb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(pemb)[
               :, None, None, :
               ]
        x = GroupNormalization()(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        x = layers.Add()([x, temb])
        x = layers.Add()([x, pemb])
        x = GroupNormalization()(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x
    return apply


def down_sample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x
    return apply


def up_sample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)
        return x

    return apply


def position_mlp(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        pemb = layers.Dense(
            units, activation=activation_fn, kernel_initializer=kernel_init(1.0)
        )(inputs)
        pemb = layers.Dense(256, kernel_initializer=kernel_init(1.0))(pemb)
        pemb = layers.Dense(128, kernel_initializer=kernel_init(1.0))(pemb)
        pemb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(pemb)
        return pemb
    return apply


def build_model(
    img_size,
    img_channels,
    first_conv_channels,
    widths,
    has_attention,
    num_res_blocks,
    norm_groups,
    interpolation="nearest",
    activation_fn=keras.activations.swish,
):
    image_input = layers.Input(shape=(img_size, img_size, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    position_input = keras.Input(shape=(2,), dtype=tf.int64, name="class_input")
    temb = layers.Embedding(1001, 128)(time_input)
    temb = layers.Dense(first_conv_channels * 4, activation=activation_fn)(temb)
    pemb = layers.Embedding(101, 20)(position_input)
    pemb = layers.Flatten()(pemb)
    pemb = layers.Dense(first_conv_channels * 4, activation=activation_fn)(pemb)
    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(image_input)
    skips = [x]
    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = residual_block(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb, pemb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)

        if widths[i] != widths[-1]:
            x = down_sample(widths[i])(x)
            skips.append(x)
    # MiddleBlock
    x = residual_block(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb, pemb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = residual_block(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb, pemb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = GroupNormalization()(x)
    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = residual_block(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb, pemb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
        if i != 0:
            x = up_sample(widths[i], interpolation=interpolation)(x)
    x = GroupNormalization()(x)
    x = activation_fn(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation=activation_fn, kernel_initializer=kernel_init(0.0))(x)
    x = layers.Dense(128, activation=activation_fn, kernel_initializer=kernel_init(0.0))(x)
    x = layers.Dense(64, kernel_initializer=kernel_init(0.0))(x)
    x = layers.Reshape((8, 8))(x)
    return keras.Model([image_input, time_input, position_input], x, name="unet")
