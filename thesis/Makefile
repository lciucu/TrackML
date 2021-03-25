LATEX    = latex
PDFLATEX = pdflatex
#BIBTEX   = bibtex
BIBTEX   = biber
DVIPS    = dvips

BASENAME = LuizaCiucuMScThesisTrackML

default: testpdflatex

testlatex:
	latex  ${BASENAME}
	latex  ${BASENAME}
	bibtex ${BASENAME}
	latex  ${BASENAME}
	latex  ${BASENAME}
	dvipdf -sPAPERSIZE=a4 -dPDFSETTINGS=/prepress ${BASENAME}

testpdflatex:
	$(PDFLATEX) ${BASENAME}
	$(PDFLATEX) ${BASENAME}
	$(BIBTEX)   ${BASENAME}
	$(PDFLATEX) $(BASENAME)
	$(PDFLATEX) ${BASENAME}

#
# standard Latex targets
#

%.dvi:	%.tex 
	$(LATEX) $<

%.bbl:	%.tex *.bib
	$(LATEX) $*
	$(BIBTEX) $*

%.ps:	%.dvi
	$(DVIPS) $< -o $@

%.pdf:	%.tex
	$(PDFLATEX) $<

.PHONY: clean

clean:
	rm -f *.aux *.log *.bbl *-blx.bib *.blg *.brf *.cb *.ind *.idx *.ilg  \
	      *.inx *.dvi *.toc *.out *~ ~* spellTmp \
		*.bcf *.lof *.lot *.run.xml *.synctex\(busy\)
	rm -f ${BASENAME}.pdf
	rm -f ${BASENAME}.synctex.gz*
	rm -f input/*.aux
	rm -f input/*~


