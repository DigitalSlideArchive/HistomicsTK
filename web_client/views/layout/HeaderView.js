import View from '../View';
import HeaderUserView from './HeaderUserView';

import headerTemplate from '../../templates/layout/header.pug';

var HeaderView = View.extend({
    events: {
    },

    render: function () {
        this.$el.html(headerTemplate());

        this.$('a[title]').tooltip({
            placement: 'bottom',
            delay: {show: 300}
        });

        new HeaderUserView({
            el: this.$('.h-current-user-wrapper'),
            parentView: this
        }).render();
    }
});

export default HeaderView;
