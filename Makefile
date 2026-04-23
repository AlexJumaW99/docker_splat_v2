# Makefile — shortcut wrappers for the docker_splat_v2 compose stack.
#
# All targets delegate to `docker compose` + smoke_test.sh. Nothing here
# is load-bearing; the Dockerfile and docker-compose.yml are the source
# of truth. If you prefer raw compose commands (e.g. for unusual flag
# combinations) they still work — this file just saves keystrokes on the
# common paths.
#
# Disaster-recovery scenario (image + container both gone):
#
#   git clone https://github.com/AlexJumaW99/docker_splat_v2.git
#   cd docker_splat_v2
#   make fresh       # build image → create container → run smoke tests
#
# Everyday use:
#
#   make up          # start container (builds image if missing)
#   make shell       # bash into /workspace
#   make test        # run full smoke suite
#   make down        # stop + remove container (named volumes survive)
#
# Run `make` or `make help` for the list.

.DEFAULT_GOAL := help
.PHONY: help build up shell down test fresh

SERVICE   := nerfstudio
CONTAINER := splat-ns-v2

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "; printf "Available targets:\n"} \
	      /^[a-zA-Z_-]+:.*?## / {printf "  %-8s  %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build (or rebuild) the nerfstudio image.
	docker compose build $(SERVICE)

up:  ## Create and start the container in detached mode.
	docker compose up -d $(SERVICE)

shell:  ## Drop into an interactive bash shell at /workspace.
	docker compose exec $(SERVICE) bash

down:  ## Stop and remove the container (named volumes survive).
	docker compose down

test:  ## Run smoke_test.sh against the running container.
	./smoke_test.sh

fresh: down build up test  ## Full rebuild from scratch: down → build → up → test.
	@echo "=== fresh rebuild complete ==="
