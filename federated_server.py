class FederatedServer:
    def __init__(self):
        """
        Initialize the Federated Server with an empty global model.
        """
        self.global_model_params = None

    def receive_updates(self, local_updates):
        """
        Receive model updates from local clients and aggregate them to update the global model.
        :param local_updates: List of model parameter dictionaries from clients
        """
        print(f"Server: Received {len(local_updates)} client updates.")
        if not self.global_model_params:
            # Initialize the global model to the first client's updates if empty
            self.global_model_params = local_updates[0]
            print("Server: Initialized global model parameters.")
        else:
            # Perform FedAvg aggregation of received updates
            for key in self.global_model_params.keys():
                self.global_model_params[key] = sum([update[key] for update in local_updates]) / len(local_updates)
            print("Server: Updated global model parameters.")

    def send_global_model(self):
        """
        Send the global model parameters to clients.
        """
        print("Server: Sending global model to clients.")
        return self.global_model_params
