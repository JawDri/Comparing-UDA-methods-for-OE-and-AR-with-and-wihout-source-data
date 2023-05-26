CoWA-JMDS Method

In CoWA.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.

In SWD.ipynb file choose the rate of data poison (0%, 5% or 10%).

Change the number of features in net.py file (len(FEATURES) = 9 for OE and len(FEATURES), line 16.

Change the task in train.py and eval.py (class_num) depending of the task that you want to test, example: ##number of classes##=3 in case of task with 3 labels.

TO TEST COWA-JMDS w/o mixup: 
!python image_source.py
!python image_target_CoWA.py  --batch_size 8 --alpha 0.0

TO TEST COWA-JMDS with mixup: 
!python image_source.py
!python image_target_CoWA.py  --batch_size 8 --alpha 1.0
