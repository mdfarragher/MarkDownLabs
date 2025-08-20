const _do_decrypt = function (encrypted, password) {
    let key = CryptoJS.enc.Utf8.parse(password);
    let iv = CryptoJS.enc.Utf8.parse(password.substr(16));
    let decrypted_data = CryptoJS.AES.decrypt(encrypted, key, {
        iv: iv,
        mode: CryptoJS.mode.CBC,
        padding: CryptoJS.pad.Pkcs7
    });
    return decrypted_data.toString(CryptoJS.enc.Utf8);
};

const _click_handler = function (element) {
    let parent = document.querySelector(".hugo-encryptor-container");
    let encrypted = parent.querySelector(".hugo-encryptor-cipher-text").innerText;
    let password = parent.querySelector(".hugo-encryptor-input").value;
    password = CryptoJS.MD5(password).toString();
    let decrypted = "";
    try {
        decrypted = _do_decrypt(encrypted, password);
    } catch (err) {
        console.error(err);
        alert("I'm sorry but the access key is incorrect.");
        return;
    }
    if (!decrypted.includes("--- DON'T MODIFY THIS LINE ---")) {
        alert("I'm sorry but the access key is incorrect..");
        return;
    }
    let storage = localStorage;
    let key = _get_storagekey(location.pathname);
    storage.setItem(key, password);
    parent.innerHTML = decrypted;
}

document.addEventListener('DOMContentLoaded', () => {

    // get password
    let key = _get_storagekey(location.pathname);
    let password = _get_password(key);
 
    // decrypt content
    let parent = document.querySelector(".hugo-encryptor-container");
    let decrypted = null;
    if (password) {
        let encrypted = parent.querySelector(".hugo-encryptor-cipher-text").innerText;
        try {
            decrypted = _do_decrypt(encrypted, password);
        } catch (err) {
            decrypted = null;
        }
    }
    if (decrypted)
        parent.innerHTML = decrypted;
    else {
        // reveal password prompt
        parent.querySelector(".hugo-encryptor-prompt").classList.remove("d-none");
        parent.querySelector(".hugo-encryptor-form").classList.add("d-flex");
        parent.querySelector(".hugo-encryptor-form").classList.remove("d-none");
    }
});
