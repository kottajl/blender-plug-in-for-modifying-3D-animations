from abc import ABC, abstractmethod

class GeneralInterface(ABC):

    @abstractmethod
    def check_frames_range() -> tuple[bool, str]:

        '''
        Check if the frames range is valid to generate propely frames in current model.
        '''

        pass

    @abstractmethod
    def infer_anim() -> tuple[list, list]:

        '''
        Infer the animation data.
        '''

        pass

# GeneralInterface
