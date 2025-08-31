function simulate_transition_matrices_manual
clc; clear;

%% ======================= Config =======================
p       = 0.1;        % channel crossover probability
STEPS   = 5e5;        % recorded transitions
BURN_IN = 5e3;        % warmup
rng(12345);           % reproducibility

fprintf('State order: [(0,2),(0,1),(0,0),(1,0),(2,0)]\n');
fprintf('p-value: %.3f\n', p);
% try for another example with m=2, trellis1 = [5 7] trellis2=[7 3]
%% =================== Trellises (poly2trellis) ===================
% We still construct trellises (K=2 -> memory=1 -> 2 states),
% but we compute branch labels manually to avoid bit-packing ambiguity.
trellis1 = poly2trellis(2, [1 3]);  %#ok<NASGU> % Code-1: (1, 1+D)
trellis2 = poly2trellis(2, [2 3]);  %#ok<NASGU> % Code-2: (D, D+1)
% (They aren't used below, but kept to show poly2trellis setup for extensibility.)

%% =================== Scenarios ===================
% 4) Random, Code-1 -> Code-1 
T4 = simulate_one(p, STEPS, BURN_IN, ...
                  1, 1, ...         % tx_code_id, rx_code_id
                  true, true);      % normalize_coset, random_input
disp(' Random, Code-1→Code-1');
print_matrix(T4);

% 6) Random, Code-2 -> Code-1 
T6 = simulate_one(p, STEPS, BURN_IN, ...
                  2, 1, ...         % tx_code_id, rx_code_id
                  false, true);     % normalize_coset, random_input
disp(' Random, Code-2→Code-1');
print_matrix(T6);

end

%% ============================= Simulation ==============================
% tx_code_id ∈ {1,2}, rx_code_id ∈ {1,2}
% normalize_coset: XOR received with noiseless label before Viterbi if true
% random_input:    Bernoulli(0.5) inputs if true; else all-zero
function T = simulate_one(p, STEPS, BURN_IN, tx_code_id, rx_code_id, normalize_coset, random_input)

STATE_LIST = [0 2; 0 1; 0 0; 1 0; 2 0];  % rows are (M0,M1)
counts = zeros(5,5);

% Relative metric (M0,M1) with min==0; encoder memory bit prev = u_{t-1}
rel = [0 0];
prev_state_bit = 0;  % for K=2, state equals previous input bit

% Burn-in
for t = 1:BURN_IN
    u = rand_bit(random_input);
    y0  = branch_bits_manual(tx_code_id, prev_state_bit, u);   % noiseless TX label (2 bits)
    r   = bsc_pair(y0, p);                                     % add BSC noise
    r_use = xor_pair(r, y0, normalize_coset);                  % coset-normalize?
    rel = viterbi_step_manual(rx_code_id, rel, r_use);         % update via RX code
    prev_state_bit = u;
end

% Main
for t = 1:STEPS
    u = rand_bit(random_input);
    y0  = branch_bits_manual(tx_code_id, prev_state_bit, u);
    r   = bsc_pair(y0, p);
    r_use = xor_pair(r, y0, normalize_coset);

    i = state_index(rel, STATE_LIST);
    rel_next = viterbi_step_manual(rx_code_id, rel, r_use);
    j = state_index(rel_next, STATE_LIST);

    counts(i,j) = counts(i,j) + 1;
    rel = rel_next;
    prev_state_bit = u;
end

% Row-normalize to get transition matrix
rowSums = sum(counts,2); rowSums(rowSums==0) = 1;
T = counts ./ rowSums;

end

%% =================== Viterbi 1-step metric update ===================
% Decoder uses its own code's branch labels (manual, consistent with Python)
function rel_next = viterbi_step_manual(rx_code_id, rel, r_use)
M0 = rel(1); M1 = rel(2);
next_metrics_abs = zeros(1,2); % for next_state = 0,1

for next_state = 0:1
    cand = zeros(1,2); % from prev_state 0 and 1
    for prev_state = 0:1
        prev_metric = (prev_state==0)*M0 + (prev_state==1)*M1;
        y_hat = branch_bits_manual(rx_code_id, prev_state, next_state); % hypothesized label
        d = hamming2(y_hat, r_use);
        cand(prev_state+1) = prev_metric + d;
    end
    next_metrics_abs(next_state+1) = min(cand);
end

mmin = min(next_metrics_abs);
rel_next = [next_metrics_abs(1)-mmin, next_metrics_abs(2)-mmin];

% Must be one of the five allowed states
allowed = [0 2; 0 1; 0 0; 1 0; 2 0];
if ~ismember(rel_next, allowed, 'rows')
    error('Unexpected relative state: [%d %d]', rel_next(1), rel_next(2));
end
end

%% =================== Manual branch labels (same as Python) ===================
% Code-1 (1, 1+D):        y = (u_t,         u_t xor u_{t-1})
% Code-2 (D, D+1):        y = (u_{t-1},     u_t xor u_{t-1})
function y = branch_bits_manual(code_id, prev_state_bit, input_bit)
if code_id == 1
    y = [input_bit, bitxor(input_bit, prev_state_bit)];
else
    y = [prev_state_bit, bitxor(input_bit, prev_state_bit)];
end
end

%% =================== Small utilities ===================
function b = rand_bit(random_input)
if random_input
    b = rand < 0.5;
else
    b = 0;
end
end

function r = bsc_pair(bits2, p)
r = [xor(bits2(1), rand < p), xor(bits2(2), rand < p)];
end

function z = xor_pair(a, b, do_xor)
if do_xor
    z = [xor(a(1), b(1)), xor(a(2), b(2))];
else
    z = a;
end
end

function d = hamming2(a, b)
d = (a(1)~=b(1)) + (a(2)~=b(2));
end

function idx = state_index(rel, STATE_LIST)
for k = 1:5
    if all(rel == STATE_LIST(k,:)), idx = k; return; end
end
error('State not found');
end

function print_matrix(T)
S = [0 2; 0 1; 0 0; 1 0; 2 0];
for i=1:5
    row = T(i,:);
    fprintf('(%.0f, %.0f) -> [%s]\n', S(i,1), S(i,2), strjoin(compose('%.2f', row), ' '));
end
end
