QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    widget.cpp\
    sdp_ort_interface.cpp

HEADERS += \
    widget.h\
    sdp_ort_interface.h
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target



win32:CONFIG(release, debug|release): LIBS += -L$$PWD/lib/ -lsdp_opencv \
                                                 -lopencv_world490 \
                                                 -lonnxruntime -lonnxruntime_providers_cuda -lonnxruntime_providers_shared

win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/lib/ -lsdp_opencv -lopencv_world490d \
                                        -lonnxruntime -lonnxruntime_providers_cuda -lonnxruntime_providers_shared \



INCLUDEPATH += $$PWD/include
DEPENDPATH += $$PWD/include


# INCLUDEPATH += $$PWD/include
# DEPENDPATH += $$PWD/include

# win32: LIBS += -L$$PWD/lib/ -lsdp_ort_staic

# win32:!win32-g++: PRE_TARGETDEPS += $$PWD/lib/sdp_onnxrt_static.lib
# else:win32-g++: PRE_TARGETDEPS += $$PWD/lib/libsdp_onnxrt_static.a
