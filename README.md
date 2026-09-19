# 4DFlow-MRI-Image-Restoration

This project investigates the use of deep learning to improve the anatomical quality of cardiac **4D Flow MRI magnitude images**. The objective is to restore degraded 4D Flow images toward the appearance of corresponding **Cine MRI images**, which provide clearer anatomical information.

The main approach is based on **Restormer**, a Transformer architecture designed for image restoration using a self-attention mechanism. Several Restormer configurations were implemented and compared, together with a conventional **U-Net baseline**, to study the impact of model architecture and capacity on restoration performance.

The experiments were evaluated using quantitative image-quality metrics such as **PSNR, SSIM, MSE, and MAE**, as well as qualitative visual comparisons. The Restormer models performed better than U-Net. Among the tested Restormer configurations, **we chose the model that provided the most convincing qualitative restoration results** after comparing quantitative metrics and seeing that we had the same results for all models tested; we then used visual assessment to select the best model for further analysis.

This project was developed during a three-month research internship at the **Leiden University Medical Center (LUMC)** within the **Division of Image Processing (LKEB)**.
