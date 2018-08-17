all: data plots plots_extra

SAMPLE := 0 1 2 3

DIRS = Data Plots

data: Data/STFTData.npz
plots: Plots/absSTFT.png

Data/STFTData.npz: pointlens.py
	python pointlens.py 768 $@

Plots/absSTFT.png: Data/STFTData.npz pointlensplot.py
	python pointlensplot.py $< abs $@


$(shell mkdir -p $(DIRS))

