import argparse
import flwr as fl


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--server-address", default="0.0.0.0:8080")
	parser.add_argument("--num-rounds", type=int, default=5)
	parser.add_argument("--min-fit-clients", type=int, default=4)
	parser.add_argument("--min-available-clients", type=int, default=4)
	parser.add_argument("--fraction-fit", type=float, default=1.0)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	strategy = fl.server.strategy.FedAvg(
		fraction_fit=args.fraction_fit,
		min_fit_clients=args.min_fit_clients,
		min_available_clients=args.min_available_clients,
	)

	print(f"Starting Flower server on {args.server_address} for {args.num_rounds} rounds")

	fl.server.start_server(
		server_address=args.server_address,
		config=fl.server.ServerConfig(num_rounds=args.num_rounds),
		strategy=strategy,
	)