import { getCurrentUser } from 'girder/auth';

import router from '../../router';
import View from '../View';

import headerUserTemplate from '../../templates/layout/headerUser.pug';

var HeaderUserView = View.extend({
    events: {
        'click a.g-login': function () {
            girder.events.trigger('g:loginUi');
        },

        'click a.g-register': function () {
            girder.events.trigger('g:registerUi');
        },

        'click a.g-logout': function () {
            girder.restRequest({
                path: 'user/authentication',
                type: 'DELETE'
            }).done(_.bind(function () {
                girder.currentUser = null;
                girder.events.trigger('g:login');
            }, this));
        },

        'click a.g-my-settings': function () {
            router.navigate('useraccount/' + getCurrentUser().get('_id') +
                            '/info', {trigger: true});
        }
    },

    initialize: function () {
        girder.events.on('g:login', this.render, this);
    },

    render: function () {
        this.$el.html(headerUserTemplate({
            user: girder.getCurrentUser()
        }));

        if (girder.currentUser) {
            this.$('.c-portrait-wrapper').css(
                'background-image', 'url(' +
                girder.currentUser.getGravatarUrl(36) + ')');
        }
        return this;
    }

});

export default HeaderUserView;
