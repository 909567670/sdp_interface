

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/lib/ -lsdp_opencv
                                      LIBS += -L$$PWD/lib/ -lopencv_world490
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/lib/ -lsdp_opencv
                                        LIBS += -L$$PWD/lib/ -lopencv_world490


INCLUDEPATH += $$PWD/include
DEPENDPATH += $$PWD/include

