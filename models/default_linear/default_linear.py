from interface.general_interface import GeneralInterface

class DefaultLinearInterface(GeneralInterface):

    '''
    Interface for the Default Linear models.
    '''

    def check_frames_range(self, start_frame, end_frame, scene_start_frame, scene_end_frame) -> bool:
        return True
    
    # end function check_frames_range


    def infer_anim(self, anim_data, start_frame, end_frame, post_processing):   
        return None, None
    
    # end function infer_anim

# DefaultLinearInterface
