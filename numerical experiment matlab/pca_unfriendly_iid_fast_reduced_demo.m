function pca_unfriendly_iid_fast_reduced_demo()
%==========================================================================
% PCA_UNFRIENDLY_IID_FAST_REDUCED_DEMO
% -------------------------------------------------------------------------
% Purpose:
%   Same i.i.d. PCA-unfriendly planted standard-form LP family as before,
%   but with an optimized reduced-LP implementation that exposes runtime
%   speedups when d >> d*.
%
% Key lesson:
%   The generic reduced formulation
%       min (U'c)'z  s.t.  x0 + U z >= 0
%   has d inequalities and may be dense / solver-unfriendly. In the planted
%   [I,T] standard-form family, every decision-relevant direction is a pivot
%   direction delta_j = [-T(:,j); e_j], so the exact reduced LP should be
%   solved in pivot coordinates:
%       min r_S' z   s.t. T_S z <= b, z >= 0.
%   This is much smaller and keeps sparsity.
%
% This script compares:
%   FullStdLP       : original standard-form LP in d=m+nExtra variables.
%   FullPivotLP     : presolved nonbasic LP in nExtra variables.
%   OursFast        : exact reduced pivot LP in d* variables.
%   OursGenericU    : old generic formulation x0 + U z >= 0.
%   PCA             : empirical-optima PCA baseline with K=d*.
%
% Requirements:
%   MATLAB Optimization Toolbox (linprog).
%==========================================================================

clc; close all;

cfg.seed       = 20260423;
cfg.m          = 150;
cfg.nExtra     = 2850;
cfg.dstar      = 30;
cfg.Tdensity   = 0.004;   % use sparse T; larger m/nExtra can use smaller density
cfg.nTrain     = 40;
cfg.nTest      = 120;
cfg.nTrials    = 5;       % runtime benchmark can be expensive
cfg.K          = cfg.dstar;

% Same iid train/test distribution. Selected pivots are rarely attractive.
cfg.pRare      = 0.004;
cfg.negLo      = -8.0;
cfg.negHi      = -4.0;
cfg.posLo      = 0.5;
cfg.posHi      = 1.2;
cfg.unselLo    = 4.0;
cfg.unselHi    = 6.0;

% Timing and solver options.
cfg.doWarmup    = true;
cfg.useDualSimplex = true;

if exist('linprog','file') ~= 2
    error('This demo requires MATLAB linprog from Optimization Toolbox.');
end

rng(cfg.seed,'twister');

methodNames = {'FullStdLP','FullPivotLP','OursFast','OursGenericU','PCA'};
nMethod = numel(methodNames);

capture = zeros(nMethod, cfg.nTrials);
relGap  = zeros(nMethod, cfg.nTrials);
runtime = zeros(nMethod, cfg.nTrials, cfg.nTest);
learnDim = zeros(cfg.nTrials,1);
pcaRank = zeros(cfg.nTrials,1);
observedSelected = zeros(cfg.nTrials,1);

for tr = 1:cfg.nTrials
    rng(cfg.seed + tr, 'twister');
    fam = generate_planted_standardform_family(cfg);
    Ctrain = sample_iid_costs(cfg.nTrain, fam, cfg);
    Ctest  = sample_iid_costs(cfg.nTest,  fam, cfg);

    % In this family the prior C explicitly identifies the possible pivot
    % directions: selected pivots have prior interval crossing zero, while
    % unselected pivots have strictly positive prior intervals. This is what
    % Algorithm-1 FI would certify. We use that learned basis here to isolate
    % the online runtime effect.
    selectedByPrior = find(fam.lbExtra < 0 & fam.ubExtra > 0);
    Ulearn = fam.DeltaAll(:, selectedByPrior);
    learnDim(tr) = size(Ulearn,2);

    % PCA from empirical training optima, K=dstar.
    XtrainOpt = zeros(cfg.nTrain, fam.d);
    obs = false(numel(fam.selected),1);
    for i = 1:cfg.nTrain
        [xOpt, ~] = solve_full_pivot_lp(Ctrain(i,:)', fam);
        XtrainOpt(i,:) = xOpt';
        activeExtra = find(xOpt(fam.m + fam.selected) > 1e-8);
        obs(activeExtra) = true;
    end
    observedSelected(tr) = sum(obs);
    [Upca, xpcaAnchor, pcaRank(tr)] = pca_basis_from_training_optima(XtrainOpt, cfg.K);

    baseVals = zeros(cfg.nTest,1);
    fullVals = zeros(cfg.nTest,1);
    vals = zeros(nMethod, cfg.nTest);

    % Warm-up reduces one-time solver overhead in timings.
    if cfg.doWarmup
        c0 = Ctest(1,:)';
        solve_full_standard_lp(c0, fam);
        solve_full_pivot_lp(c0, fam);
        solve_reduced_pivot_fast(c0, selectedByPrior, fam);
        solve_reduced_generic(c0, Ulearn, fam.x0, fam);
        solve_reduced_generic(c0, Upca, xpcaAnchor, fam);
    end

    for i = 1:cfg.nTest
        c = Ctest(i,:)';
        baseVals(i) = c' * fam.x0;

        tic; [~, vals(1,i)] = solve_full_standard_lp(c, fam); runtime(1,tr,i)=toc;
        fullVals(i) = vals(1,i);

        tic; [~, vals(2,i)] = solve_full_pivot_lp(c, fam); runtime(2,tr,i)=toc;
        tic; [~, vals(3,i)] = solve_reduced_pivot_fast(c, selectedByPrior, fam); runtime(3,tr,i)=toc;
        tic; [~, vals(4,i)] = solve_reduced_generic(c, Ulearn, fam.x0, fam); runtime(4,tr,i)=toc;
        tic; [~, vals(5,i)] = solve_reduced_generic(c, Upca, xpcaAnchor, fam); runtime(5,tr,i)=toc;
    end

    fullImprove = max(baseVals - fullVals, 0);
    denom = max(sum(fullImprove), 1e-10);
    for mtd = 1:nMethod
        methodImprove = max(baseVals - vals(mtd,:)', 0);
        capture(mtd,tr) = max(0, min(1, sum(methodImprove) / denom));
        relGap(mtd,tr) = mean(max(0, vals(mtd,:)' - fullVals) ./ max(1, abs(fullVals)));
    end

    fprintf(['trial %d/%d | d=%d, m=%d, d*=%d | observed selected=%d/%d | ', ...
             'cap OursFast=%.3f, PCA=%.3f | time fullStd=%.3g, oursFast=%.3g, oldGeneric=%.3g\n'], ...
        tr, cfg.nTrials, fam.d, fam.m, learnDim(tr), observedSelected(tr), cfg.dstar, ...
        capture(3,tr), capture(5,tr), mean(runtime(1,tr,:)), mean(runtime(3,tr,:)), mean(runtime(4,tr,:)));
end

outDir = prepare_results_dir('pca_unfriendly_fast_reduced_results');

meanCap = mean(capture,2); ciCap = 1.645 * std(capture,0,2) / sqrt(cfg.nTrials);
fig1 = figure('Color','w'); hold on; box on; grid on;
bar(meanCap); errorbar(1:nMethod, meanCap, ciCap, '.k', 'LineWidth',1.2);
set(gca,'XTick',1:nMethod,'XTickLabel',methodNames,'XTickLabelRotation',25);
ylabel('Improvement capture ratio'); ylim([0,1.05]);
title('Objective quality: PCA-unfriendly iid standard-form LP');

meanGap = mean(relGap,2); ciGap = 1.645 * std(relGap,0,2) / sqrt(cfg.nTrials);
fig2 = figure('Color','w'); hold on; box on; grid on;
bar(meanGap); errorbar(1:nMethod, meanGap, ciGap, '.k', 'LineWidth',1.2);
set(gca,'XTick',1:nMethod,'XTickLabel',methodNames,'XTickLabelRotation',25);
ylabel('Mean relative objective gap');
title('Relative objective gap');

meanTimeTrial = squeeze(mean(runtime,3))';
meanTime = mean(meanTimeTrial,1); ciTime = 1.645 * std(meanTimeTrial,0,1) / sqrt(cfg.nTrials);
fig3 = figure('Color','w'); hold on; box on; grid on;
bar(meanTime); errorbar(1:nMethod, meanTime, ciTime, '.k', 'LineWidth',1.2);
set(gca,'YScale','log');
set(gca,'XTick',1:nMethod,'XTickLabel',methodNames,'XTickLabelRotation',25);
ylabel('Mean solve time per test LP (seconds)');
title('Runtime: fast pivot reduced LP vs old generic U-form');

fig4 = figure('Color','w'); box on; grid on;
bar([mean(learnDim), mean(pcaRank), mean(observedSelected), cfg.dstar]);
set(gca,'XTick',1:4,'XTickLabel',{'learned dim','PCA rank','observed selected','true d*'},'XTickLabelRotation',20);
ylabel('Count / dimension');
title('Diagnostics');

save_figure(fig1, fullfile(outDir,'improvement_capture_bars.png'));
save_figure(fig2, fullfile(outDir,'relative_gap_bars.png'));
save_figure(fig3, fullfile(outDir,'runtime_bars_fast_vs_generic.png'));
save_figure(fig4, fullfile(outDir,'diagnostics_bars.png'));
save(fullfile(outDir,'summary.mat'), 'cfg','methodNames','capture','relGap','runtime','learnDim','pcaRank','observedSelected');

fprintf('\nSaved results to %s\n', outDir);
end

%==========================================================================
% Family and sampling
%==========================================================================
function fam = generate_planted_standardform_family(cfg)
m = cfg.m; nExtra = cfg.nExtra; d = m + nExtra;
T = sprand(m, nExtra, cfg.Tdensity);
T = abs(T);
% ensure each extra column has at least one nonzero; keep sparsity
for j = 1:nExtra
    if nnz(T(:,j)) == 0
        T(randi(m),j) = 0.05 + rand;
    end
end
T = sparse(T);
A = [speye(m), T];
b = 2.0 + rand(m,1);
x0 = [b; zeros(nExtra,1)];
selected = sort(randperm(nExtra, cfg.dstar));
unselected = setdiff(1:nExtra, selected);
DeltaAll = sparse(d,nExtra);
for j = 1:nExtra
    col = sparse(d,1);
    col(1:m) = -T(:,j);
    col(m+j) = 1;
    DeltaAll(:,j) = col;
end
lbExtra = zeros(nExtra,1); ubExtra = zeros(nExtra,1);
lbExtra(selected) = cfg.negLo; ubExtra(selected) = cfg.posHi;
lbExtra(unselected) = cfg.unselLo; ubExtra(unselected) = cfg.unselHi;

fam.A = A; fam.b = b; fam.T = T; fam.m = m; fam.nExtra = nExtra; fam.d = d;
fam.x0 = x0; fam.selected = selected; fam.unselected = unselected; fam.DeltaAll = DeltaAll;
fam.lbExtra = lbExtra; fam.ubExtra = ubExtra;
end

function C = sample_iid_costs(n, fam, cfg)
C = zeros(n, fam.d);
for i = 1:n
    c = zeros(fam.d,1); % basis costs are zero in this planted family
    rare = rand(numel(fam.selected),1) < cfg.pRare;
    vals = cfg.posLo + (cfg.posHi-cfg.posLo)*rand(numel(fam.selected),1);
    vals(rare) = cfg.negLo + (cfg.negHi-cfg.negLo)*rand(sum(rare),1);
    c(fam.m + fam.selected) = vals;
    c(fam.m + fam.unselected) = cfg.unselLo + (cfg.unselHi-cfg.unselLo)*rand(numel(fam.unselected),1);
    C(i,:) = c';
end
end

%==========================================================================
% Solvers
%==========================================================================
function [x,val] = solve_full_standard_lp(c, fam)
opts = make_linprog_options();
[x,val,exitflag] = linprog(c, [], [], fam.A, fam.b, zeros(fam.d,1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(x)
    error('Full standard-form LP failed.');
end
end

function [x,val] = solve_full_pivot_lp(c, fam)
% Equivalent full problem in nonbasic variables y:
% x_B = b - T y >= 0, y>=0.
r = c(fam.m+1:end) - fam.T' * c(1:fam.m);
opts = make_linprog_options();
[y,obj,exitflag] = linprog(r, fam.T, fam.b, [], [], zeros(fam.nExtra,1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(y)
    y = zeros(fam.nExtra,1); obj = 0;
end
x = [fam.b - fam.T*y; y];
val = c(1:fam.m)'*fam.b + obj;
end

function [x,val] = solve_reduced_pivot_fast(c, selectedCols, fam)
% Fast exact reduced LP in pivot coordinates for selected columns S:
% min r_S'z s.t. T_S z <= b, z>=0.
if isempty(selectedCols)
    x = fam.x0; val = c' * x; return;
end
S = selectedCols(:)';
r = c(fam.m + S) - fam.T(:,S)' * c(1:fam.m);
opts = make_linprog_options();
[z,obj,exitflag] = linprog(r, fam.T(:,S), fam.b, [], [], zeros(numel(S),1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(z)
    z = zeros(numel(S),1); obj = 0;
end
x = fam.x0;
x(1:fam.m) = fam.b - fam.T(:,S)*z;
x(fam.m + S) = z;
val = c(1:fam.m)'*fam.b + obj;
end

function [x,val] = solve_reduced_generic(c, U, xAnchor, fam)
% Old generic representation: min (U'c)'z s.t. xAnchor + U z >=0 and AUz=0.
if isempty(U), x=xAnchor; val=c'*x; return; end
opts = make_linprog_options();
r = U' * c;
Aineq = -U; bineq = xAnchor;
AU = fam.A * U;
if norm(AU,'fro') <= 1e-8
    Aeq=[]; beq=[];
else
    Aeq=AU; beq=fam.b - fam.A*xAnchor;
end
[z,~,exitflag] = linprog(r, Aineq, bineq, Aeq, beq, -inf(size(U,2),1), [], opts); %#ok<ASGLU>
if exitflag <= 0 || isempty(z), z = zeros(size(U,2),1); end
x = xAnchor + U*z; val = c'*x;
end

function [U, xbar, effRank] = pca_basis_from_training_optima(Xtrain, K)
xbar = mean(Xtrain,1)'; Xc = Xtrain - mean(Xtrain,1);
[~,S,V] = svd(Xc,'econ'); sv=diag(S);
if isempty(sv), effRank=0; else, effRank=sum(sv>1e-9*max(1,sv(1))); end
r = min([K, effRank, size(V,2)]);
if r==0, U=zeros(size(Xtrain,2),0); else, U=V(:,1:r); end
end

%==========================================================================
% Helpers
%==========================================================================
function opts = make_linprog_options()
if exist('optimoptions','file') == 2
    opts = optimoptions('linprog','Display','none','Algorithm','dual-simplex','Presolve','on');
else
    opts = optimset('Display','off'); %#ok<OPTIMSET>
end
end

function save_figure(fig, filename)
set(fig,'Color','w');
if exist('exportgraphics','file') == 2
    exportgraphics(fig,filename,'Resolution',200);
else
    saveas(fig,filename);
end
end

function outDir = prepare_results_dir(prefix)
stamp = datestr(now,'yyyymmdd_HHMMSS'); outDir = fullfile(pwd,[prefix '_' stamp]);
if ~exist(outDir,'dir'), mkdir(outDir); end
end
