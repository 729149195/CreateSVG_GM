## 测试：
#### Windows:

* `python -m venv test_env `建新环境
* `Set-ExecutionPolicy RemoteSigned`开启权限
* `.\test_env\Scripts\activate`激活环境
* `deactivate`退出测试环境
* `rmdir /s /q test_env`删除测试环境(CMD)
* `Remove-Item test_env -Recurse -Force`删除测试环境(PowerShell)


#### Unix/Linux or MacOS:

* `python -m venv test_env `建新环境
* `Set-ExecutionPolicy RemoteSigned`开启权限
* `source test_env/bin/activate`激活环境
* `deactivate`退出测试环境
* `rm -rf test_env`删除测试环境

## 安装依赖：
`pip install -r requirements.txt`


GitHub URL: https://github.com/729149195/CreateSVG_GM