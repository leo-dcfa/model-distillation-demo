# Serve the marimo notebooks locally.
#
# Ports are fixed (not random) so the SSH port-forward command from another
# machine is deterministic. Override on the command line if a port is taken,
# e.g. `make explore EXPLORE_PORT=3000`.

VIZ_PORT     ?= 2718
EXPLORE_PORT ?= 2719
HOST         ?= 127.0.0.1

.PHONY: help viz viz-run explore explore-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

viz: ## Edit the distillation-mechanism notebook (sliders)
	uv run marimo edit notebooks/distillation_viz.py --headless --host $(HOST) --port $(VIZ_PORT)

viz-run: ## Serve the distillation-mechanism notebook read-only
	uv run marimo run notebooks/distillation_viz.py --headless --host $(HOST) --port $(VIZ_PORT)

explore: ## Edit the mechanistic-interp notebook (TransformerLens)
	uv run marimo edit notebooks/explore_distilled_models.py --headless --host $(HOST) --port $(EXPLORE_PORT)

explore-run: ## Serve the mechanistic-interp notebook read-only
	uv run marimo run notebooks/explore_distilled_models.py --headless --host $(HOST) --port $(EXPLORE_PORT)
