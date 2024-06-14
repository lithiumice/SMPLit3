from decord import VideoReader



class CustomVideoReader(VideoReader):
    def __init__(self, path, target_fps=None):
        """
        @target_fps: if -1 or None, use origin video fps
        """
        path = str(path)
        self.path = path
        super().__init__(path)
        self.original_fps = super().get_avg_fps()
        # self.target_fps = target_fps if target_fps else self.original_fps
        if (
            target_fps == -1 or target_fps is None
        ):
            self.target_fps = self.original_fps
        else:
            self.target_fps = target_fps

    def get_avg_fps(self):
        return self.target_fps
    
    def __len__(self):
        return int((super().__len__()/self.original_fps)*self.target_fps)

    def get_target_frame_pos(self, target_fps_pos):
        return int((target_fps_pos / self.target_fps) * self.original_fps)

    def __getitem__(self, index):
        target_fps_frame_pos = self.get_target_frame_pos(index)
        return super().__getitem__(target_fps_frame_pos)

    def get_batch(self, indices):
        target_fps_frame_positions = [self.get_target_frame_pos(index) for index in indices]
        return super().get_batch(target_fps_frame_positions)




