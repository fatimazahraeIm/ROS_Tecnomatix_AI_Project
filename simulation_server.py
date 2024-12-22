import xmlrpc.server
import comtypes.client as comtypes

class SimulationServer:
    def __init__(self):
        self.plant_simulation = None

    def loadModel(self, path_file):
        self.plant_simulation = comtypes.CreateObject("Tecnomatix.PlantSimulation.RemoteControl")
        self.plant_simulation.loadModel(path_file)
        return "Model loaded"

    def setVisible(self, value):
        if self.plant_simulation:
            self.plant_simulation.setVisible(value)
            return "Visibility set"
        return "Not connected"

    def setValue(self, ref, value):
        if self.plant_simulation:
            self.plant_simulation.setValue(ref, value)
            return "Value set"
        return "Not connected"

    def startSimulation(self, model):
        if self.plant_simulation:
            self.plant_simulation.startSimulation(model)
            return "Simulation started"
        return "Not connected"

    def IsSimulationRunning(self):
        if self.plant_simulation:
            return self.plant_simulation.IsSimulationRunning()
        return False

    def getValue(self, ref):
        if self.plant_simulation:
            return self.plant_simulation.getValue(ref)
        return None

    def resetSimulation(self, model):
        if self.plant_simulation:
            self.plant_simulation.resetSimulation(model)
            return "Simulation reset"
        return "Not connected"

server = xmlrpc.server.SimpleXMLRPCServer(("0.0.0.0", 8000))
server.register_instance(SimulationServer())
print("Server is running...")
server.serve_forever()