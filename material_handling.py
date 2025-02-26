
# ============================================================================
# IMPORTS
# ============================================================================

from plant import Plant
import numpy as np
import rospy
import time
from std_msgs.msg import String

# ============================================================================


class Material_Handling(Plant):

    def __init__(self, method):
        self.pub = rospy.Publisher(
            'plant_simulation_topic', String, queue_size=10)
        Plant.__init__(self, method)

    def get_file_name_plant(self):
        return "your-path/MaterialHandling.spp"

    def update(self, data):
        # Convert numpy.int64 to native Python int
        data = [int(d) for d in data]

        self.connect.setValue(".Models.Modele.attente", data[0])
        self.connect.setValue(".Models.Modele.stock", data[1])
        self.connect.setValue(".Models.Modele.nombrevoyages", data[2])
        self.connect.startSimulation(".Models.Modele")

        a = np.zeros(9)
        b = np.zeros(20)
        c = np.zeros(20)

        while self.connect.IsSimulationRunning():
            time.sleep(0.1)

        for g in range(1, 10):
            a[g-1] = self.connect.getValue(
                ".Models.Modele.transport[2,%s]" % (g))
        for h in range(1, 21):
            b[h-1] = self.connect.getValue(
                ".Models.Modele.buffer[3,%s]" % (h))
            c[h-1] = self.connect.getValue(
                ".Models.Modele.sortie1[2,%s]" % (h))
        d = np.sum(a)
        e = np.sum(b)
        f = np.sum(c)
        r = d * 0.3 + e * 0.3 + f * 0.4

        self.connect.resetSimulation(".Models.Modele")

        # topic publish
        resultado_str = "YES"
        rospy.loginfo(resultado_str)
        self.pub.publish(resultado_str)
        return r

    def process_simulation(self):
        if (self.connection()):
            self.connect.setVisible(True)
            self.method.process()
            self.method.plot()
