###############################################################################
#  Output mode base class — video/audio output interface
###############################################################################

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
from numpy.typing import NDArray
import numpy as np

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


class BaseOutput(ABC):
    """
    Abstract base class for output transport modes.

    Implementers must provide:
    1. start(): start the output
    2. push_video_frame(): push a video frame
    3. push_audio_frame(): push an audio frame
    4. stop(): close the output
    """

    def __init__(self, opt=None, parent: Optional['BaseAvatar'] = None, **kwargs):
        self.opt = opt
        self.parent = parent

    @abstractmethod
    def start(self) -> None:
        """Start the output channel"""
        ...

    @abstractmethod
    def push_video_frame(self, frame) -> None:
        """Push a video frame"""
        ...

    @abstractmethod
    def push_audio_frame(self, frame:NDArray[np.int16], eventpoint=None) -> None:
        """Push an audio frame"""
        ...


    def get_buffer_size(self) -> int:
        """Get the number of backlogged frames in the underlying send queue, used to throttle the engine"""
        return 0

    @abstractmethod
    def stop(self) -> None:
        """Close the output channel"""
        ...
