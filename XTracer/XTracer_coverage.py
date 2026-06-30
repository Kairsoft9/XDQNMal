# -*- coding: utf-8 -*-
import json
import sys
import threading
import time
import frida
import subprocess
import os
from copy import copy
from PyQt5.QtWidgets import *
import XT_config
from multiprocessing import Process

sys.setrecursionlimit(5000)

# 选择检测文件路径
# feature_path_ey = r'H:\samples\dataset_x\malice'
feature_path_lx = r'H:\samples\dataset_x\benign'

chooseApkPath = feature_path_lx

storage_path = chooseApkPath + '/0000_storage_coverage.yml'
coverage_path = chooseApkPath + '/0000_coverage.yml'
time_spent_path = chooseApkPath + '/0000_time_spent.yml'
featurePath = chooseApkPath + r'/feature_c'
if not os.path.exists(featurePath):
    os.makedirs(featurePath)

# 记录覆盖率和运行时间结果
coverage = XT_config.config(coverage_path)
if coverage.data is None:
    coverage.data={}
time_spent = XT_config.config(time_spent_path)
if time_spent.data is None:
    time_spent.data={}
# 记录检测结果
storage = XT_config.config(storage_path)
if storage.data is None:
    storage.data = {'0000_success_num': 0,
                    '0000_fail_num': 0}

XT = None
scripts = []
loadingAPK = ''
hookSuccess = False
running_state = 0


class XTracerData:
    def __init__(self, tracer=None):
        super(XTracerData, self).__init__()
        self.tracer = tracer

    def stop(self):
        global scripts
        for s in copy(scripts):
            try:
                s.unload()
            except frida.InvalidOperationError as e:
                print(e)
                scripts.remove(s)
            else:
                scripts.remove(s)
        print('[G] success unload script')

    def clean(self):
        self.tracer.thread_map = {}

    def export(self):
        global loadingAPK, hookSuccess
        fpath, fname = os.path.split(loadingAPK)
        api = fname.split('.apk')[0]
        jobfile = featurePath + '//' + api + '.txt'
        thread_map = self.tracer.thread_map.copy()
        if thread_map != {}:
            json.dump(thread_map, open(jobfile, 'w', encoding='utf-8'))
            hookSuccess = True
            print('[G] success get jsonLog')
        else:
            print('[G] jsonLog is None')


class XTracer:
    def __init__(self, ):
        global XT
        XT = self
        self.application_label = None
        self.packageName = None
        self.tracer = QApplication(sys.argv)
        self.trace_data = XTracerData(self)
        self.thread_map = {}
        self.hookComplete = 'false'
        self.singleTrace()
        sys.exit()

    def method_entry(self, tid, clazz, method, args):
        data_list = []
        if tid in self.thread_map:
            data_list = self.thread_map[tid]
        data_list.append([clazz, method, args])
        self.thread_map[tid] = data_list

    def log(self, text):
        text = time.strftime('%Y-%m-%d %H:%M:%S:  [*] ', time.localtime(time.time())) + text
        print(text)

    def singleTrace(self):
        global loadingAPK, running_state
        # running_state = 1
        apkPaths = getApkPath(chooseApkPath)
        apk_index = 0
        while True:
            loadingAPK = apkPaths[apk_index]
            fpath, fname = os.path.split(loadingAPK)
            apk_name = fname.split('.apk')[0]
            if apk_name in storage.data:
                apk_index += 1
                if apk_index == len(apkPaths):
                    print('当前文件夹已无apk')
                    return
                continue
            print('[A] ------------------ start ---------------------------')
            start_time = time.time()
            if self.appTrace(apk_name):
                storage.data[apk_name] = 'success'
                storage.data['0000_success_num'] += 1
                print("[J] Trace Success ", loadingAPK)
                # 获取运行时间
                end_time = time.time()
                execution_time = end_time - start_time
                time_spent.data[apk_name]=execution_time
                time_spent.saveData()
            else:
                storage.data[apk_name] = 'fail'
                storage.data['0000_fail_num'] += 1
                print("[J] Trace Fail ", loadingAPK)
            storage.saveData()
            print('    ------------------ end -----------------------------')
            # running_state = 2

    def appTrace(self,apk_name):
        global hookSuccess
        print('[A] apk_Path:' + str(loadingAPK))
        self.application_label = getPackageLabel()
        self.packageName = getPackageName()
        packageActivity = getPackageActivity()
        # 判断文件是否存在
        if self.packageName is None or packageActivity is None:
            return False
        if apkInstall():
            # 清空上个程序运行残留
            self.trace_data.clean()
            # 运行安装apk
            if runApk(self.packageName, packageActivity):
                # 运行追踪脚本
                if self.runTrace():
                    # 运行monkey和覆盖率监测
                    runMonkey(self.packageName,apk_name)
                    # 获取日志
                    self.getJsonLog()
                    # 获取日志后卸载脚本
                    self.trace_data.stop()
                # 无论运行是否成功，都结束运行
                stopApk(self.packageName)
            # 最后卸载app
            apkUninstall(self.packageName)
        if hookSuccess:
            # 初始化hookSuccess
            hookSuccess = False
            return True
        return False

    def runTrace(self):
        print('[E] ------------------ hooking ------------------------')
        print('[E]',time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        # threading.Thread(target=self.start_trace).start()
        # while True:
        #     if 'true' in self.hookComplete:
        #         print('[E] success hook')
        #         # 重新初始化hookComplete
        #         self.hookComplete = 'false'
        #         return True
        #     if 'fail' in self.hookComplete:
        #         print('[E] fail hook')
        #         # 重新初始化hookComplete
        #         self.hookComplete = 'false'
        #         return False
        #     time.sleep(5)
        if self.start_trace():
            print('[E] success hook')
            return True
        else:
            print('[E] fail hook')
            return False

    def start_trace(self):
        global scripts
        hook_process_num = 0
        def _attach(pid,trycount):
            if trycount>0:
                try:
                    session = device.attach(pid)
                    session.enable_child_gating()
                    source = open('XTracer.js', 'r', encoding='utf-8').read().replace('{hook_list}', str(hook_list()))
                    script = session.create_script(source)
                    script.on("message", self.FridaReceive)
                    script.load()
                    scripts.append(script)
                    print('[E] success hooked process:', process)
                    return True
                except Exception as e:
                    printRed("An error occurred:"+ str(e))
                    return _attach(pid, trycount-1)
            else:
                return False


        def _on_child_added(child):
            print('[E] hook child_process:', child)
            _attach(child.pid,2)

        device = frida.get_usb_device()
        # device.on("child-added", _on_child_added)
        for process in device.enumerate_processes():
            if self.packageName:
                if self.packageName in process.name:
                    print('[E] start hooking  process:', process)
                    if _attach(process.pid,2):
                        hook_process_num += 1
                        continue
            if self.application_label:
                if self.application_label in process.name:
                    print('[E] start hooking process:', process)
                    if _attach(process.pid,2):
                        hook_process_num += 1
        # 若无attach成功则判断hook失败
        # time.sleep(5)
        if hook_process_num == 0:
            return False
        else:
            return True

    def FridaReceive(self, message, data):
        if message['type'] == 'send':
            if message['payload'][:10] == 'XTracer:::':
                packet = json.loads(message['payload'][10:])
                cmd = packet['cmd']
                mdata = packet['data']
                if cmd == 'log':
                    # 接收hook完成日志
                    if 'Hook Complete' in mdata:
                        self.hookComplete = 'true'
                elif cmd == 'enter':
                    tid, cls, method, args = mdata
                    XT.method_entry(tid, cls, method, args)
                # elif cmd == 'exit':
                #     tid, retval = data
                #     XT.method_exit(tid, retval)
        else:
            print(message['stack'])

    def getJsonLog(self):
        print('[G] ------------------ get jsonLog --------------------')
        self.trace_data.export()


def hook_list():
    extend_list = [
        'android.content.Intent/$init',
        'android.content.Intent/putExtra',
        'android.content.Intent/setAction',
        'android.support.v4.app.ActivityCompat/requestPermissions',
        'androidx.core.app.ActivityCompat/requestPermissions'
    ]
    with open("source/hook_list_479.csv") as f:
        hookList = [row.split(',')[0] for row in f]
    for item in extend_list:
        hookList.append(item)
    del (hookList[0])
    return hookList


def getApkPath(path):
    apkPaths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if '.apk' in file:
                apkPath = path + '\\' + file
                apkPaths.append(apkPath)
    return apkPaths


def runCMD(command):
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, encoding="utf-8").stdout
    return result


def getPackageLabel():
    adbReturn = runCMD('aapt dump badging ' + loadingAPK + '| findstr application-label:')
    if 'application-label' in adbReturn:
        labelName = adbReturn.split("application-label:'")[1].split("'")[0]
        print('[B] labelName: ' + labelName)
        return labelName
    else:
        return None


def getPackageName():
    adbReturn = runCMD('aapt dump badging ' + loadingAPK + '| findstr package')
    if 'package: name' in adbReturn:
        package_name = adbReturn.split("package: name='")[1].split("' versionCode")[0]
        print('[B] package_name: ' + package_name)
        return package_name
    else:
        printRed('[B] fail get package_name--apkPath:' + loadingAPK)
        return None


def getPackageActivity():
    adbReturn = runCMD('aapt dump badging ' + loadingAPK + '| findstr activity')
    if 'activity: name' in adbReturn:
        activityName = adbReturn.split("activity: name='")[1].split("'  label")[0]
        print('[B] mainActivityName: ' + activityName)
        return activityName
    else:
        printRed('[B] fail get activityName--apkPath:' + loadingAPK)
        return None


def apkInstall():
    print('[C] ------------------ APK installing -----------------')
    command = 'adb install -r ' + loadingAPK
    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)
    try:
        adbReturn = proc.communicate(180)
        for readLine in adbReturn:
            if 'Success' in str(readLine):
                print('[C] APK install success')
                return True
        return False
    except subprocess.TimeoutExpired:
        printRed('[C] APK install fail--apkPath:' + loadingAPK)
        return False


def runApk(package, packageActivity):
    print('[D] ------------------ APK running --------------------')
    command = 'adb shell am start -W -n ' + package + '/' + packageActivity
    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)
    try:
        adbReturn = proc.communicate(timeout=60)
        if 'Complete' in str(adbReturn):
            print('[D] start running')
            # 关闭线程
            proc.kill()
            time.sleep(1)
            return True
    except subprocess.TimeoutExpired:
        proc.kill()
        printRed('[D] fail run--TimeoutExpired')
        return False


def runMonkey(package,apk_name):
    print('[F] ------------------ monkey running -----------------')
    # command = 'adb shell CLASSPATH=/sdcard/monkey.jar:/sdcard/framework.jar exec app_process /system/bin tv.panda.test.monkey.Monkey -p ' + package + ' --uiautomatormix --pct-reset 0 --pct-rotation 0 --running-minutes 2 -v'
    command = 'adb shell CLASSPATH=/sdcard/monkeyq.jar:/sdcard/framework.jar:/sdcard/fastbot-thirdpart.jar exec app_process /system/bin com.android.commands.monkey.Monkey -p' + package + ' --agent reuseq --running-minutes 2 --throttle 500'
    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)
    try:
        adbReturn = proc.communicate(timeout=300)
        for readline in adbReturn:
            if 'Activity of Coverage' in str(readline):
                coveragenum=str(readline).split('Coverage:')[1].split('%')[0]
                coverage.data[apk_name]=coveragenum
                coverage.saveData()
                print('[F] success get coverage')
                return
    except subprocess.TimeoutExpired:
        printRed('[F] fail run monkey')


def stopApk(package):
    print('[H] ------------------ app stopping -------------------')
    command = 'adb shell pm clear ' + package
    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)
    try:
        adbReturn = proc.communicate(timeout=120)
        if 'Success' in str(adbReturn):
            print('[H] success stopping')
            return
    except subprocess.TimeoutExpired:
        printRed('[H] fail stop')


def apkUninstall(package):
    print('[I] ------------------ uninstalling -------------------')
    command = 'adb uninstall ' + package
    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE)
    try:
        adbReturn = proc.communicate(timeout=120)
        for readLine in adbReturn:
            if 'Success' in str(readLine):
                print('[I] uninstalled')
                return
    except subprocess.TimeoutExpired:
        printRed('[I] fail uninstall')


def printRed(message):
    print("\033[1;31;48m" + message + "\033[0m")


if __name__ == '__main__':
    # hookThread = Process(target=XTracer)
    # hookThread.start()
    # while True:
    #     if running_state==0:
    #         adbReturn = subprocess.run('frida-ps -U', shell=True, stdout=subprocess.PIPE, encoding="utf-8").stdout
    #         if "Failed" in adbReturn:
    #             printRed('Frida Disconnect, Waiting Frida')
    #         if "PID" in adbReturn:
    #             hookThread = Process(target=XTracer)
    #             hookThread.start()
    #             hookThread.join()
    #     elif running_state==2:
    #         print("运行结束")
    #         break
    #     time.sleep(10)
    while True:
        adbReturn = subprocess.run('frida-ps -U', shell=True, stdout=subprocess.PIPE, encoding="utf-8").stdout
        if "Failed" in adbReturn:
            printRed('Frida Disconnect, Waiting Frida')
        if "PID" in adbReturn:
            hookThread = Process(target=XTracer)
            hookThread.start()
            hookThread.join()
        time.sleep(10)
