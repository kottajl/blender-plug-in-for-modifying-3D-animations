from abc import ABC, abstractmethod

class GeneralInterface(ABC):

    @abstractmethod
    def check_frames_range() -> tuple[bool, str]:

        '''
        Check if the frames range is valid to generate propely frames in current model.
        '''

        pass

    @abstractmethod
    def get_infer_anim_kwargs() -> list[tuple[type, str, str]]:

        '''
        Return kwargs as list with tuples (variable type, name, description).
        '''

        pass

    @abstractmethod
    def infer_anim() -> tuple[list, list]:

        '''
        Infer the animation data.
        '''

        pass

# GeneralInterface
