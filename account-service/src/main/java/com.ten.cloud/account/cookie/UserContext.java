package com.ten.cloud.account.cookie;

import com.ten.cloud.account.account.User;

public class UserContext{

    private static ThreadLocal<UserContext> tl = new ThreadLocal<>();

    private User user;

}
