const fileInput = document.querySelector('input[type="file"]');
const previewContainer = document.createElement("div");

fileInput.parentNode.appendChild(previewContainer);

fileInput.addEventListener("change", function () {
    previewContainer.innerHTML = "";

    const file = this.files[0];
    if (!file) return;

    const img = document.createElement("img");
    img.style.width = "120px";
    img.style.borderRadius = "10px";
    img.style.marginTop = "10px";

    const reader = new FileReader();
    reader.onload = function (e) {
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    const removeBtn = document.createElement("button");
    removeBtn.innerText = "Remove";
    removeBtn.style.marginLeft = "10px";

    removeBtn.onclick = () => {
        fileInput.value = "";
        previewContainer.innerHTML = "";
    };

    previewContainer.appendChild(img);
    previewContainer.appendChild(removeBtn);
});