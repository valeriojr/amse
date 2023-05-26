import enum
import socket
import struct
import subprocess

import numpy


class MessageType(enum.Enum):
    Undefined = 0
    Observation = 1
    Action = 2
    Reward = 3
    Stop = 4


class UnityEnvironment:
    def __init__(self, executable_path, host='127.0.0.1', port=65432):
        self.socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        if executable_path is not None:
            self.simulation_process = subprocess.Popen([executable_path])

        self.socket.bind((host, port))

        print(f'Listening on {host}:{port}')
        self.socket.listen()

        self.connection, self.client_address = self.socket.accept()
        print(f'Connected to environment at {self.client_address}')

    def send_message(self, message_type: MessageType, payload: bytes) -> None:
        message_header = (
                message_type.value.to_bytes(length=1, byteorder='little') +
                len(payload).to_bytes(length=4, byteorder='little')
        )
        self.connection.send(message_header)
        self.connection.send(payload)

    def receive_message(self) -> (MessageType, bytes):
        message_header = self.connection.recv(5)
        message_type = MessageType(message_header[0])
        payload_length = int.from_bytes(message_header[1:], byteorder='little')
        payload = self.connection.recv(payload_length)

        return message_type, payload

    def expect_message(self, message_type: MessageType) -> bytes:
        received_message_type, payload = MessageType.Undefined, None

        while received_message_type != message_type:
            received_message_type, payload = self.receive_message()

            if received_message_type != message_type:
                raise ValueError(f'Expected {message_type}, got {received_message_type} instead')

        return payload

    @staticmethod
    def _get_state(buffer: bytes):
        speed, angle = struct.unpack('ff', buffer[:8])
        camera = numpy.frombuffer(buffer[8:], dtype=numpy.uint8).reshape((96, 96, 3)).astype(numpy.float32) / 255.0

        return camera, speed, angle

    def get_observation(self):
        return self._get_state(self.expect_message(MessageType.Observation))

    def step(self, action):
        self.send_message(MessageType.Action, struct.pack('2i', *action))
        reward_payload = self.expect_message(MessageType.Reward)
        reward, done = struct.unpack('fb', reward_payload)
        next_state = self.get_observation()

        return next_state, reward, done

    def close(self):
        self.connection.close()
        self.simulation_process.kill()
