package com.ten.wechat.service.impl;

import com.ten.wechat.service.TestService;
import org.springframework.stereotype.Service;

@Service
public class TestServiceImpl implements TestService {
    @Override
    public String test(String content) {
        return "this is test :" + content;
    }
}
