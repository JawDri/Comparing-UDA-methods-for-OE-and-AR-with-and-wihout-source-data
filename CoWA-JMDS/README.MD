CoWA-JMDS Method

In CoWA.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.


TO TEST COWA-JMDS w/o mixup: 
!python image_source.py
!python image_target_CoWA.py  --batch_size 8 --alpha 0.0

TO TEST COWA-JMDS with mixup: 
!python image_source.py
!python image_target_CoWA.py  --batch_size 8 --alpha 1.0
