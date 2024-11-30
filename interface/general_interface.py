from abc import ABC, abstractmethod

class GeneralInterface(ABC):

    @abstractmethod
    def check_frames_range() -> tuple[bool, str]:

        '''
        Check if frame range is valid to generate propely frames in current model, returns (True, "") if yes,
        or (False, <error message>) if not.
        '''

        pass

    @abstractmethod
    def get_infer_anim_kwargs() -> list[tuple[type, str, str]]:

        '''
        Returns list with tuples (type, name, description) for every variable in kwargs for infer_anim().
        '''

        pass

    @abstractmethod
    def infer_anim() -> tuple[list, list]:

        '''
        Infer the animation data, returns (inferred positions, inferred rotations) for in-between frames.
        '''

        pass

# GeneralInterface
