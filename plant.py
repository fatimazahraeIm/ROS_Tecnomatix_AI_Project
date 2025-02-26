from abc import ABCMeta, abstractmethod
from communication_interface import Communication_Interface

class Plant(metaclass=ABCMeta):
    def __init__(self, method):
        self.method = method
        method.register(self)

    def connection(self):
        file_name = self.get_file_name_plant()
        server_url = "server_url:8000"  # Replace with the actual server URL
        self.connect = Communication_Interface(server_url)
        return self.connect.connection(file_name)

    @abstractmethod
    def get_file_name_plant(self):
        pass

    @abstractmethod
    def process_simulation(self):
        pass

    @abstractmethod
    def update(self, data):
        pass
