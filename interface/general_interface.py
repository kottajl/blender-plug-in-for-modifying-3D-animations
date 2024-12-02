from abc import ABC, abstractmethod

class GeneralInterface(ABC):

    @abstractmethod
    def check_frames_range(self, start_frame: int, end_frame: int, 
                           scene_start_frame: int, scene_end_frame: int
    ) -> tuple[bool, str]:

        '''
        Check if frame range is valid to generate propely frames in current model, returns (True, "") if yes,
        or (False, <error message>) if not.
        '''

        pass

    @abstractmethod
    def get_additional_infer_params(self) -> list[tuple[type, str, str]]:
        
        '''
        Returns list with tuples (type, name, description) for every additional parameter that is required for 
        the inferring animation process. Those parameters should be later passed to infer_anim() as kwargs.
        '''

        pass

    @abstractmethod
    def infer_anim(self, anim_data, start_frame: int, end_frame: int, **kwargs) -> tuple[list, list]:

        '''
        Infer the animation data, returns (inferred positions, inferred rotations) for in-between frames.
        '''

        pass

    @abstractmethod
    def is_skeleton_supported(self, skeleton: tuple[str, int]) -> bool:

        '''
        Check if the given skeleton is supported by the model, if this function returns false frame generation
        will be aborted.
        '''

        pass

# GeneralInterface
