---
layout: default
---
<a href="images/energy_in_feature_space.png" target="_blank"><img src="images/energy_in_feature_space.png"/></a>

## Abstract

Modern machine learning models, that excel on computer vision tasks such as classification and object detection, are often overconfident in their predictions for Out-of-Distribution (OOD) examples, resulting in unpredictable behaviour for open-set environments. Recent works have demonstrated that the free energy score is an effective measure of uncertainty for OOD detection given its close relationship to the data distribution. However, despite free energy-based methods representing a significant empirical advance in OOD detection, our theoretical analysis reveals previously unexplored and inherent vulnerabilities within the free energy score formulation such that in-distribution and OOD instances can have distinct feature representations yet identical free energy scores. This phenomenon occurs when the vector direction representing the feature space difference between the in-distribution and OOD sample lies within the null space of the last layer of a neural-based classifier. To mitigate these issues, we explore lower-dimensional feature spaces to reduce the null space footprint and introduce novel regularisation to maximize the least singular value of the final linear layer, hence enhancing inter-sample free energy separation. We refer to these techniques as Free Energy Vulnerability Elimination for Robust Out-of-Distribution Detection (FEVER-OOD). Our experiments show that FEVER-OOD techniques achieve state of the art OOD detection in Imagenet-100, with average OOD false positive rate (at 95% true positive rate) of 35.83% when used with the baseline Dream-OOD model.

[//]: # (<a href="images/small_architecture.png" target="_blank"><img src="images/small_architecture.png"/></a>)

## Results

<div class="slideshow-container">
  <div class="mySlides fade">
    <div class="numbertext">1 / 4</div>
    <div class="card">
        <a href="images/mscoco_vos.jpg" target="_blank"><img class='card-img' src="images/mscoco_vos.jpg"/></a>
        <div class="card-container">
            <h4>ID:PASCAL VOC, OOD:MSCOCO</h4>
        </div>
    </div>
  </div>

  <div class="mySlides fade">
    <div class="numbertext">2 / 4</div>
    <div class="card">
        <a href="images/mscoco_ffs.jpg" target="_blank"><img class='card-img' src="images/mscoco_ffs.jpg"/></a>
        <div class="card-container">
            <h4>ID:PASCAL VOC, OOD:MSCOCO</h4>
        </div>
    </div>
  </div>

<div class="mySlides fade">
    <div class="numbertext">3 / 4</div>
    <div class="card">
        <a href="images/openimages_vos.jpg" target="_blank"><img class='card-img' src="images/openimages_vos.jpg"/></a>
        <div class="card-container">
            <h4>ID:PASCAL VOC, OOD:OpenImages</h4>
        </div>
    </div>
  </div>

<div class="mySlides fade">
    <div class="numbertext">4 / 4</div>
    <div class="card">
        <a href="images/openimages_ffs.jpg" target="_blank"><img class='card-img' src="images/openimages_ffs.jpg"/></a>
        <div class="card-container">
            <h4>ID:PASCAL VOC, OOD:OpenImages</h4>
        </div>
    </div>
  </div>

  <!-- Next and previous buttons -->
  <a class="prev" onclick="plusSlides(-1)">&#10094;</a>
  <a class="next" onclick="plusSlides(1)">&#10095;</a>
</div>
<br>

<!-- The dots/circles -->
<div style="text-align:center">
  <span class="dot" onclick="currentSlide(1)"></span>
  <span class="dot" onclick="currentSlide(2)"></span>
  <span class="dot" onclick="currentSlide(3)"></span>
  <span class="dot" onclick="currentSlide(4)"></span>
</div>


## Citation
    {% raw %}
    @article{isaac-medina24fever-ood, 
    author = {Isaac-Medina, B.K.S. and Che, M. and Gaus, Y.F.A. and Akcay, S. and Breckon, T.P.}, 
    title = {FEVER-OOD: Free Energy Vulnerability Elimination for Robust Out-of-Distribution Detection}, 
    journal={arXiv preprint arXiv:2412.01596}, 
    year = {2024}, 
    month = {December}, }
    {% endraw %}