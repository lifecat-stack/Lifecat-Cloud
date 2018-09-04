package com.ten.wechat.controller;

import com.ten.wechat.service.TestService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class BaseController {

    @Autowired
    private TestService testService;

    @RequestMapping(value = "/test")
    @ResponseBody
    public String test(){
        System.out.println("load");
        return testService.test("test service exe");
    }
}
