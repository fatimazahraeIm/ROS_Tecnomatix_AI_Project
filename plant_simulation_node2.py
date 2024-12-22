# ============================================================================
# IMPORTS
# ============================================================================

import rospy
from material_handling import Material_Handling
from rl_method_4 import RL_Method_4

# ============================================================================


def plant_simulation_node():
    rospy.init_node('plant_simulation_node', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    pretrained_model = "model-12-499.pt"  # Replace with the path to your pretrained model
    method = RL_Method_4(pretrained_model)
    plant = Material_Handling(method)
    plant.process_simulation()

    rate.sleep()


if __name__ == '__main__':
    try:
        plant_simulation_node()
    except rospy.ROSInterruptException:
        pass