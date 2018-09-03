package com.ten.wechat;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration;

@SpringBootApplication
@EnableAutoConfiguration
class WechatWebApplication {

    public static void main(String[] args) {
        SpringApplication.run(WechatWebApplication.class, args);
    }

}
