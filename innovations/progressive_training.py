class ProgressiveResolutionScheduler:
    """
    Schedules resolution progression during training.
    
    Starts at a fraction of the target resolution and gradually increases
    it across specified milestones, reducing early-stage computational cost
    and improving convergence stability.
    """

    def __init__(self, schedule_str="0.5,0.75,1.0", milestones_str="100000,300000"):
        self.scales = [float(x) for x in schedule_str.split(',')]
        self.milestones = [int(x) for x in milestones_str.split(',')]
        assert len(self.scales) == len(self.milestones) + 1, \
            "Schedule should have one more scale than milestones."

    def get_scale(self, iteration):
        for i, milestone in enumerate(self.milestones):
            if iteration < milestone:
                return self.scales[i]
        return self.scales[-1]
