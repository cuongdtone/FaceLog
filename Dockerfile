# Author: Cuong Tran
FROM facelog_2

RUN mkdir /app
WORKDIR /app

COPY . .


# Minimize image size
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

ENV QT_X11_NO_MITSHM=1

CMD ["bash"]
