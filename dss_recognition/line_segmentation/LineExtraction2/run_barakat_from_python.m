function [result, Labels, linesMask, newLines] = run_barakat_from_python(filename)
    addpath("anigauss\");
    addpath("Binarization\Itay\matlab\")
    addpath("Code\");
    addpath("EvolutionMap\");
    addpath("gco-v3.0\matlab\");
    addpath("matlab_bgl\");
    addpath("Multi_Skew_Code\");
    addpath("SLMtools\");
    
    I = imread(filename);
    bin = ~I;
    [result, Labels, linesMask, newLines] = ExtractLines(I, bin);		% Extract the lines, linesMask = intermediate line results for debugging.
%     imshow(label2rgb(newLines));
end