import { getCurrentUser } from 'girder/auth';
import GirderHeaderUserView from 'girder/views/layout/HeaderUserView';

import headerUserTemplate from '../../templates/layout/headerUser.pug';

var HeaderUserView = GirderHeaderUserView.extend({
    render() {
        this.$el.html(headerUserTemplate({
            user: getCurrentUser()
        }));
        return this;
    }
});

export default HeaderUserView;
