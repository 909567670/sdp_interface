#ifndef WIDGET_H
#define WIDGET_H

#include <QTextCodec>
#include <QWidget>

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    inline std::string _T(const char* str) {
        return QTextCodec::codecForName("GBK")->fromUnicode(str).toStdString();
    }
};
#endif // WIDGET_H
