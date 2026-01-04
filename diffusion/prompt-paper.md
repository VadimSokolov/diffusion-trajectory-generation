Now write a document diffusion.qmd (Quarto for Mermaid support) that later on will become part of tehcnical paper to be submitted to IEEE ITS journal (https://ieee-itss.org/pub/t-its/) this paper will present results from multiple models, including this one. diffusion.md must contain:
1. Neural network Architecture description. With mermaid diagram
2. Loss function
3. Training algorithm
3a. Entire pipeline with Quarto diagram
4. Generaton process. Describe all "knobs" and how they can be used to adjust generated trajectories. 
5. Justify every detail of the architecture, trainig, loss and generation process. Specifically talk about what we've learned from failed attempts. 
6. Results of simulations (sample trajectories, histograms, heat plots, tables) Describe those results
7. Suggesitons on future improvemen of the model
9. How this model can be used to geenrate trajectories by conditioning on vehicle type and traffic/road. 

When you talk about modeling, add references. I am using `/Users/vsokolov/Dropbox/prj/svtrip/paper3/svtrip-ai.bib` bibtex file, use references from there and zotero. Update bibtex file as needed. 

diffusion.qmd will be merged later into IEEE ITS paper. Look inside v* folders and *.md files to learn about why different assumptions and modeling choices were made, document them carefully 