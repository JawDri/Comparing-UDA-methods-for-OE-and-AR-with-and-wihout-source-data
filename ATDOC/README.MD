ATDOC Method 
In ATDOC.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.



TO TEST ATDOC-NC: !python demo_uda.py --pl atdoc_nc 

TO TEST ATDOC-nA: !python demo_uda.py --pl atdoc_na

TO TEST ATDOC-NC combined with CDAN+E: !python demo_uda.py --pl atdoc_nc --method CDANE

TO TEST ATDOC-nA combined with CDAN+E: !python demo_uda.py --pl atdoc_na --method CDANE
