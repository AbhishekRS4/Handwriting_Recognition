function [result, Labels, linesMask, newLines] = run_barakat_from_python
    addpath("anigauss\");
    addpath("Binarization\Itay\matlab\")
    addpath("Code\");
    addpath("EvolutionMap\");
    addpath("gco-v3.0\matlab\");
    addpath("matlab_bgl\");
    addpath("Multi_Skew_Code\");
    addpath("SLMtools\");
    I = imread('P513-Fg001-R-C01-R01-binarized.jpg');
    bin = ~I;
    [result, Labels, linesMask, newLines] = ExtractLines(I, bin);		% Extract the lines, linesMask = intermediate line results for debugging.
%     imshow(label2rgb(newLines));
end