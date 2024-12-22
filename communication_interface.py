import xmlrpc.client

class Communication_Interface:
    def __init__(self, server_url):
        self.server = xmlrpc.client.ServerProxy(f"http://{server_url}")
        self.is_connected = False

    def connection(self, path_file):
        try:
            self.server.loadModel(path_file)
            print("The connection was successful")
            self.is_connected = True
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def setVisible(self, value):
        if self.is_connected:
            self.server.setVisible(value)
        else:
            print("Not connected")

    def setValue(self, ref, value):
        if self.is_connected:
            self.server.setValue(ref, value)
        else:
            print("Not connected")

    def startSimulation(self, model):
        if self.is_connected:
            self.server.startSimulation(model)
        else:
            print("Not connected")

    def IsSimulationRunning(self):
        if self.is_connected:
            return self.server.IsSimulationRunning()
        else:
            print("Not connected")
            return False

    def getValue(self, ref):
        if self.is_connected:
            return self.server.getValue(ref)
        else:
            print("Not connected")
            return None

    def resetSimulation(self, model):
        if self.is_connected:
            self.server.resetSimulation(model)
        else:
            print("Not connected")