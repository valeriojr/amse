import traceback

import numpy
import tensorflow as tf
import tqdm as tqdm
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

import unity_environment as unity


def build_actor(learning_rate):
    camera = layers.Input(shape=(96, 96, 3), name='Camera')
    speed = layers.Input(shape=(1,), name='Speed')
    angle = layers.Input(shape=(1,), name='Steering angle')

    x = camera
    x = layers.Conv2D(input_shape=(96, 96, 3), filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, speed, angle])
    x = layers.Dense(units=32, activation='relu')(x)
    x = layers.Dense(units=16, activation='relu')(x)
    x = layers.Dense(units=8, activation='relu')(x)

    gas = layers.Dense(units=3, activation='softmax')(x)
    steering = layers.Dense(units=3, activation='softmax')(x)

    actor = models.Model(inputs=[camera, speed, angle],
                         outputs=[gas, steering])
    actor.optimizer = optimizers.adam_v2.Adam(learning_rate=learning_rate)

    return actor


def build_critic(learning_rate):
    camera = layers.Input(shape=(96, 96, 3), name='Camera')
    speed = layers.Input(shape=(1,), name='Speed')
    angle = layers.Input(shape=(1,), name='Steering angle')

    x = camera
    x = layers.Conv2D(input_shape=(96, 96, 3), filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, speed, angle])
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    x = layers.Dense(units=32, activation='relu')(x)

    value = layers.Dense(units=1)(x)

    critic = models.Model(inputs=[camera, speed, angle], outputs=[value])
    critic.optimizer = optimizers.adam_v2.Adam(learning_rate=learning_rate)

    return critic


def build_state(camera, speed, angle):
    return [
        numpy.expand_dims(camera, axis=0),
        numpy.expand_dims(speed, axis=0),
        numpy.expand_dims(angle, axis=0),
    ]


gamma = 0.99
checkpoint_threshold = 10
env = unity.UnityEnvironment('../carlos/Build/carlos.exe')
# env = unity.UnityEnvironment(None)
actor = build_actor(learning_rate=0.00001)
critic = build_critic(learning_rate=0.001)

try:
    for episode in tqdm.tqdm(range(1000)):
        camera, speed, angle = env.get_observation()

        while True:
            [gas_prob, steering_prob] = actor(build_state(camera, speed, angle))
            gas = numpy.random.choice([-1, 0, 1], p=numpy.squeeze(gas_prob))
            steering = numpy.random.choice([-1, 0, 1], p=numpy.squeeze(steering_prob))

            (next_camera, next_speed, next_angle), r, done = env.step((gas, steering))

            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                [gas_prob, steering_prob] = actor(build_state(camera, speed, angle))

                value = critic(build_state(camera, speed, angle))
                next_value = critic(build_state(next_camera, next_speed, next_angle))

                advantage = r + gamma * next_value - value

                actor_loss = -(tf.math.log(gas_prob[0, gas]) + tf.math.log(steering_prob[0, steering])) * advantage
                critic_loss = advantage ** 2

            actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
            critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
            critic.optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

            if done:
                break

            camera = next_camera
            speed = next_speed
            angle = next_angle

        if (episode + 1) % checkpoint_threshold == 0:
            actor.save('models/actor.h5')
            critic.save('models/critic.h5')

except Exception as e:
    env.send_message(unity.MessageType.Stop, bytes())
    env.close()
    print(traceback.format_exc())
