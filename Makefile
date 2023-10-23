# -------------------------------------   Clear logs and cache  -------------------------------------
clear-logs:
	rm -rf logs/*

clear-slurm-logs:
	find . -maxdepth 1 -type f -name "slurm-*.out" -delete

clear-wandb-cache:
	rm -rf wandb/*

clear-all:
	make clear-logs
	make clear-slurm-logs
	make clear-wandb-cache
