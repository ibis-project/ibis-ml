```{=html}
<ul id="choose-your-tool" class="nav nav-tabs" role="tablist">
  <h3 class="no-anchor">Choose your tool</h3>
  <li class="nav-item" role="presentation">
    <a class="nav-link" href="scikit-learn.html">
      <img src="images/scikit-learn-logo.svg">sklearn
    </a>
  </li>
  <li class="nav-item" role="presentation">
    <a class="nav-link" href="xgboost.html">
      <img src="images/xgboost-logo.png">XGBoost
    </a>
  </li>
  <li class="nav-item" role="presentation">
    <a class="nav-link" href="pytorch.html">
      <img src="images/pytorch-logo.svg">PyTorch
    </a>
  </li>
</ul>

<script type="text/javascript">
document.addEventListener("DOMContentLoaded", function() {
  // get file name
  const filename = window.location.pathname.split("/").slice(-1)[0];

  // latch active
  const toolLinks = window.document.querySelectorAll("#choose-your-tool a");
  for (const tool of toolLinks) {
    if (tool.href.endsWith(filename)) {
      tool.classList.add("active");
      break;
    }
  }

  // save in local storage
  window.localStorage.setItem("tutorialTool", filename);
});

</script>
```
