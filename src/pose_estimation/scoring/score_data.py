from __future__ import annotations
class ScoreData():
    # Explicitly for Angle Score
    r_shoulder_l_shoulder_l_elbow = []
    l_shoulder_l_elbow_l_wrist = []

    l_shoulder_r_shoulder_r_elbow = []
    r_shoulder_r_elbow_r_wrist = []

    r_hip_l_hip_l_knee = []
    l_hip_l_knee_l_ankle = []

    l_hip_r_hip_r_knee = []
    r_hip_r_knee_r_ankle = []

    def store_score(self, scores: list[float]):
        self.r_shoulder_l_shoulder_l_elbow.append(scores[0])
        self.l_shoulder_l_elbow_l_wrist.append(scores[1])
        self.l_shoulder_r_shoulder_r_elbow.append(scores[2])
        self.r_shoulder_r_elbow_r_wrist.append(scores[3])
        self.r_hip_l_hip_l_knee.append(scores[4])
        self.l_hip_l_knee_l_ankle.append(scores[5])
        self.l_hip_r_hip_r_knee.append(scores[6])
        self.r_hip_r_knee_r_ankle.append(scores[7])

    def calculate_average(self):
        avg_r_shoulder_l_shoulder_l_elbow = []
        i = 0
        sum = 0
        while(i < len(self.r_shoulder_l_shoulder_l_elbow)):
            if i%29 == 0:
                print("i = ",i)
                print("sum= ", sum)
                avg_r_shoulder_l_shoulder_l_elbow.append(sum/30)
                sum = 0
            else:
                sum += self.r_shoulder_l_shoulder_l_elbow[i]
            i+=1
        return avg_r_shoulder_l_shoulder_l_elbow
    
    def create_map(self, )->dict:
        return{
            'r_shoulder_l_shoulder_l_elbow': self.r_shoulder_l_shoulder_l_elbow,
            'l_shoulder_l_elbow_l_wrist': self.l_shoulder_l_elbow_l_wrist,

            'l_shoulder_r_shoulder_r_elbow': self.l_shoulder_r_shoulder_r_elbow,
            'r_shoulder_r_elbow_r_wrist' : self.r_shoulder_r_elbow_r_wrist,
            
            'r_hip_l_hip_l_knee' : self.r_hip_l_hip_l_knee,
            'l_hip_l_knee_l_ankle' : self.l_hip_l_knee_l_ankle,

            'l_hip_r_hip_r_knee': self.l_hip_r_hip_r_knee,
            'r_hip_r_knee_r_ankle': self.r_hip_r_knee_r_ankle
        }


    def calculate_average_all(self):
        map_data = self.create_map()
        for key in map_data:
            avg = []
            i = 0
            sum = 0
            while i < len(getattr(self, key)):
                if (i + 1) % 30 == 0:  # Check if we have processed 30 elements
                    avg.append(round((sum / 30), 2))
                    sum = 0
                else:
                    sum += getattr(self, key)[i]
                i += 1
            # If there are remaining elements, calculate average for them
            if sum > 0:
                avg.append(round((sum / (i % 30)), 2))
            setattr(self, key, avg)

        final_map = self.create_map()
        return final_map


