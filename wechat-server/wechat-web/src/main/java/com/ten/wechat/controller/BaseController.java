package com.ten.wechat.controller;

import com.ten.wechat.service.TestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class BaseController {

    @Autowired
    private TestService testService;

    @RequestMapping(value = "/test")
    public String test(){
        return testService.test("test service exe");
    }
}
