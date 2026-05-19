run-antmaze:
    cd src && uv run train_inverse_dynamics.py --center_reward --actor_use_full_encoder --batch_size 1024
run-pointmaze:
    cd src && uv run train_inverse_dynamics.py --env_name pointmaze-medium-navigate-v0 \
     --actor_use_full_encoder \
     --batch_size 1024 \
     --phi_dim 32 \
     --psi_dim 32 \
     --enc_hidden_dim 128 \
     --enc_n_hidden 2 \
     --pred_hidden_dim 128 \
     --pred_n_hidden 2 \
     --actor_use_full_encoder \
     --bc_coeff 0.0 \
     --eval_every 1000 \
     --log_every 100