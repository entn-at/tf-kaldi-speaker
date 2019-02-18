clear

step_per_epoch = 30000;
num_epoch = 40;
step = 1:step_per_epoch*num_epoch;

lambda_min = 0;
lambda_base = 1000;
gamma = 0.00001;
lambda_power = 5;

lambda = max(lambda_min, lambda_base * (1 + gamma * step).^(-lambda_power));
fa = 1.0 ./ (1.0 + lambda);
figure
plot(step, lambda);
xlim([0 step_per_epoch*num_epoch])
ylim([0 20])
figure();
plot(step, fa);