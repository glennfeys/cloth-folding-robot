"""This module communicates with Unity to set the correct
correct training mode, either multi-stage either single stage
"""
import uuid
from mlagents_envs.side_channel import SideChannel, IncomingMessage, OutgoingMessage


class ModeChannel(SideChannel):
    """This Unity ML agents Side Channel is used to communicate
    in which mode the training should happen.
    """

    def __init__(self) -> None:
        super().__init__(uuid.UUID("df1a3b81-8285-417e-a81f-ead61ff54977"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        raise NotImplementedError

    def send_string(self, data: str) -> None:
        """This message is used to send the string
        that sets the training mode in Unity

        :param data: Either the string "Multi" or "Single"
        :type data: str
        """
        msg = OutgoingMessage()
        msg.write_string(data)
        super().queue_message_to_send(msg)
